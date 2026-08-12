//! Tests for the user-credential lifecycle: create, alter, rename, drop, and
//! durable reload, including cache coherence and password verification.

use std::sync::Arc;

use zyron_auth::{HeapAuthStorage, PasswordCredential, SecurityManager, User, UserId};
use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_storage::{DiskManager, DiskManagerConfig};

async fn make_storage(dir: &std::path::Path) -> Arc<HeapAuthStorage> {
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(dir.to_path_buf()))
        .await
        .expect("disk"),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    Arc::new(HeapAuthStorage::new(disk, pool).expect("auth storage"))
}

fn user(name: &str, hash: Option<String>) -> User {
    User {
        id: UserId(0),
        name: name.to_string(),
        password_hash: hash,
        scram_secret: None,
        md5_credential: None,
        api_key_prefix: None,
        api_key_hash: None,
        totp_secret: None,
        connection_limit: -1,
        valid_until: None,
        locked: false,
        locked_at: None,
        locked_reason: None,
        created_at: 1,
        superuser: false,
        can_login: true,
    }
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

#[test]
fn create_alter_rename_drop_lifecycle() {
    let tmp = tempfile::TempDir::new().unwrap();
    block_on(async {
        let storage = make_storage(tmp.path()).await;
        let sm = SecurityManager::new(storage)
            .await
            .expect("security manager");

        // Create with a hashed password.
        let hash = PasswordCredential::from_plaintext("secret")
            .unwrap()
            .as_stored()
            .to_string();
        sm.create_user(&user("alice", Some(hash.clone())))
            .await
            .expect("create");

        let got = sm.lookup_user("alice").expect("alice exists");
        assert!(got.id.0 > 0, "id allocated");
        assert_eq!(sm.password_cache.get(&"alice".to_string()), Some(hash));
        let cred = PasswordCredential::from_stored(got.password_hash.clone().unwrap());
        assert!(cred.verify("secret").unwrap(), "password verifies");
        assert!(!cred.verify("wrong").unwrap(), "wrong password rejected");

        // Duplicate create is rejected.
        assert!(sm.create_user(&user("alice", None)).await.is_err());

        // Update the password.
        let new_hash = PasswordCredential::from_plaintext("newpw")
            .unwrap()
            .as_stored()
            .to_string();
        let mut updated = got.clone();
        updated.password_hash = Some(new_hash.clone());
        sm.update_user(&updated).await.expect("update");
        assert_eq!(
            sm.password_cache.get(&"alice".to_string()),
            Some(new_hash.clone())
        );

        // Rename re-keys every cache and preserves the id.
        let id_before = sm.lookup_user("alice").unwrap().id;
        sm.rename_user("alice", "bob").await.expect("rename");
        assert!(sm.lookup_user("alice").is_none());
        let bob = sm.lookup_user("bob").expect("bob exists");
        assert_eq!(bob.id, id_before, "id preserved across rename");
        assert!(sm.password_cache.get(&"alice".to_string()).is_none());
        assert_eq!(sm.password_cache.get(&"bob".to_string()), Some(new_hash));
        assert_eq!(sm.user_id_cache.get(&"bob".to_string()), Some(id_before));

        // Drop removes the record everywhere; a second drop is a no-op.
        assert!(sm.drop_user("bob").await.expect("drop"));
        assert!(sm.lookup_user("bob").is_none());
        assert!(sm.password_cache.get(&"bob".to_string()).is_none());
        assert!(!sm.drop_user("bob").await.expect("second drop"));
    });
}

#[test]
fn users_persist_and_id_counter_recovers_across_reload() {
    let tmp = tempfile::TempDir::new().unwrap();
    block_on(async {
        let storage = make_storage(tmp.path()).await;
        let assigned_id;
        {
            let sm = SecurityManager::new(Arc::clone(&storage) as Arc<_>)
                .await
                .expect("sm1");
            let hash = PasswordCredential::from_plaintext("pw")
                .unwrap()
                .as_stored()
                .to_string();
            sm.create_user(&user("carol", Some(hash))).await.unwrap();
            assigned_id = sm.lookup_user("carol").unwrap().id;
        }

        // A fresh manager over the same storage loads the user and resumes the
        // id counter beyond the max persisted id.
        let sm2 = SecurityManager::new(Arc::clone(&storage) as Arc<_>)
            .await
            .expect("sm2");
        let carol = sm2.lookup_user("carol").expect("carol loaded");
        assert_eq!(carol.id, assigned_id);

        sm2.create_user(&user("dave", None)).await.unwrap();
        let dave = sm2.lookup_user("dave").unwrap();
        assert!(dave.id.0 > assigned_id.0, "new id does not collide");
    });
}

#[test]
fn rename_into_existing_name_is_rejected() {
    let tmp = tempfile::TempDir::new().unwrap();
    block_on(async {
        let storage = make_storage(tmp.path()).await;
        let sm = SecurityManager::new(storage).await.unwrap();
        sm.create_user(&user("u1", None)).await.unwrap();
        sm.create_user(&user("u2", None)).await.unwrap();
        assert!(
            sm.rename_user("u1", "u2").await.is_err(),
            "collision rejected"
        );
        assert!(
            sm.lookup_user("u1").is_some(),
            "source intact after failed rename"
        );
    });
}
