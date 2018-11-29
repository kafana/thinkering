import unittest


from wenchain.core import Transaction, Wallet, TransactionPool
from wenchain.core.common import (
    MINING_REWARD,
    REWARD_TRANSACTION_ADDRESS,
    REWARD_TRANSACTION_SIGNATURE)


TEST_WALLET_BALANCE = 500


class TransactionTests(unittest.TestCase):

    test_password = b' '
    test_private_key_pem = (
        b'-----BEGIN ENCRYPTED PRIVATE KEY-----\n'
        b'MIIBHDBXBgkqhkiG9w0BBQ0wSjApBgkqhkiG9w0BBQwwHAQILrSYZjgYyooCAggA\n'
        b'MAwGCCqGSIb3DQIJBQAwHQYJYIZIAWUDBAEqBBADXEa7N8SLjVU4LNo5c/R8BIHA\n'
        b'k1iuYs2U0kuTjery/6etVINw1ftBUTo3t9yaXz1UAftF0OqknWY2jB5w++UwmojU\n'
        b'0qiX0rE67hzxo1/MSbjPT+ZuwjYvewBRaic/76H2P2h/+0eVA/FD0tW3err13goM\n'
        b'TZhHOvNmDl0bu6yh4g9OadF0sXeovpQched/t1uBFAj59gtDcmY1T+P1X6uNK3Oz\n'
        b'FCnmuheritHg9GJyrXRVMnTtj/Do/uAcZwDwzNjltfeS46PePOeWkrnVTkbIJ+86\n'
        b'-----END ENCRYPTED PRIVATE KEY-----\n')
    test_public_key_pem = (
        b'-----BEGIN PUBLIC KEY-----\n'
        b'MHYwEAYHKoZIzj0CAQYFK4EEACIDYgAE3y0ROVUSFZExyZnZAQtGr2+s8nmCrlZM\n'
        b'3hngFDhDCM2nuEK/KDlo3WqIRC1iM+D9F6WHKgpTMnkusbgj8dvNyBMP8PglISyv\n'
        b'01L2Gw2/nifhOf+/nIWiWelUHe89VSjt\n'
        b'-----END PUBLIC KEY-----\n')
    test_recipient_address1 = (
        '14ddr3551cce621e8a42203ef370d2b8f2428d8910e00b4491784872e286fb086be358283dccadf80214d78bd'
        'c6a369cae091f2c01be7a68be4e94a778955a51a40002430a0004018b250601020d3ec8468a270600103650')
    test_recipient_address2 = (
        '24ddr3552cce621e8a42203ef370d2b8f2428d8910e00b4491784872e286fb086be358283dccadf80214d78bd'
        'c6a369cae091f2c01be7a68be4e94a778955a51a40002430a0004018b250601020d3ec8468a270600103650')

    def test_create_transaction(self):
        amount = 50
        wallet = Wallet(self.test_password, self.test_private_key_pem, self.test_public_key_pem,
                        balance=TEST_WALLET_BALANCE)
        transaction = Transaction.create_transaction(wallet, self.test_recipient_address1, amount)
        self.assertEqual(transaction.outputs[0].address, wallet.public_key_hex)
        self.assertEqual(transaction.outputs[0].amount, wallet.balance - amount)
        self.assertEqual(transaction.outputs[1].address, self.test_recipient_address1)
        self.assertEqual(transaction.outputs[1].amount, amount)
        self.assertEqual(transaction.input.amount, wallet.balance)

    def test_transaction_verification(self):
        amount = 50
        wallet = Wallet(self.test_password, self.test_private_key_pem, self.test_public_key_pem,
                        balance=TEST_WALLET_BALANCE)
        transaction = Transaction.create_transaction(wallet, self.test_recipient_address1, amount)
        self.assertTrue(transaction.verify())
        transaction.outputs[0].address = 'changed public key'
        self.assertFalse(transaction.verify())

    def test_transaction_update(self):
        amount1 = 50
        wallet = Wallet(self.test_password, self.test_private_key_pem, self.test_public_key_pem,
                        balance=TEST_WALLET_BALANCE)
        transaction = Transaction.create_transaction(wallet, self.test_recipient_address1, amount1)
        self.assertTrue(transaction.verify())
        amount2 = 30
        transaction.update(wallet, self.test_recipient_address2, amount2)
        self.assertTrue(transaction.verify())
        self.assertEqual(transaction.outputs[0].amount, wallet.balance - (amount1 + amount2))
        transaction.outputs[0].amount = 10000000
        self.assertFalse(transaction.verify())

    def test_transaction_pool(self):
        pool = TransactionPool()
        wallet = Wallet(self.test_password, self.test_private_key_pem, self.test_public_key_pem,
                        balance=TEST_WALLET_BALANCE)

        amount1 = 50
        transaction1 = Transaction.create_transaction(wallet, self.test_recipient_address1, amount1)
        pool.add(transaction1)

        transaction2 = Transaction.create_transaction(wallet, self.test_recipient_address1, amount1)
        pool.add(transaction2)

        amount2 = 30
        transaction3 = Transaction.create_transaction(wallet, self.test_recipient_address1, amount2)
        transaction3.transaction_id = transaction1.transaction_id # Set tx3 id to tx1 id
        pool.add(transaction3)

        self.assertTrue(len(pool.transactions) == 2)
        self.assertEqual(pool.transactions[transaction1.transaction_id].outputs[0].address,
                         wallet.public_key_hex)
        self.assertEqual(pool.transactions[transaction1.transaction_id].outputs[0].amount,
                         wallet.balance - amount2)
        self.assertEqual(pool.transactions[transaction1.transaction_id].outputs[1].address,
                         self.test_recipient_address1)
        self.assertEqual(pool.transactions[transaction1.transaction_id].outputs[1].amount,
                         amount2)

    def test_valid_transactions(self):
        pool = TransactionPool()
        wallet = Wallet(self.test_password, self.test_private_key_pem, self.test_public_key_pem,
                        balance=TEST_WALLET_BALANCE)

        amount1 = 50
        transaction1 = Transaction.create_transaction(wallet, self.test_recipient_address1, amount1)
        pool.add(transaction1)

        transaction2 = Transaction.create_transaction(wallet, self.test_recipient_address1, amount1)
        pool.add(transaction2)

        transactions = pool.get_valid_transactions()
        self.assertTrue(len(transactions) == 2)

        # Let's corrupt balance
        transaction1.outputs[0].amount = 1000
        transactions = pool.get_valid_transactions()
        self.assertTrue(len(transactions) == 1)

        # Let's corrupt signature
        transaction2.outputs[0].address = 'corrupted-address'
        transactions = pool.get_valid_transactions()
        self.assertTrue(len(transactions) == 0)

    def test_reward_transactions(self):
        miner_wallet = Wallet(self.test_password, self.test_private_key_pem,
                              self.test_public_key_pem,
                              balance=TEST_WALLET_BALANCE)
        transaction1 = Transaction.create_reward_transaction(1, miner_wallet)
        self.assertEqual(transaction1.outputs[0].address, miner_wallet.public_key_hex)
        self.assertEqual(transaction1.outputs[0].amount, MINING_REWARD)
        self.assertEqual(transaction1.input.address, REWARD_TRANSACTION_ADDRESS)
        self.assertEqual(transaction1.input.signature, REWARD_TRANSACTION_SIGNATURE)
